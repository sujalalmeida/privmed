-- ============================================================================
-- UPDATE EXISTING USERS TABLE FOR PRIVMED
-- Run this in Supabase Dashboard → SQL Editor
-- Updates the existing users table to work with the application
-- ============================================================================

-- 1. Add missing columns that the application expects
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS full_name TEXT;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS contact_phone TEXT;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS role TEXT;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS assigned_lab_id UUID;

-- 2. Update role column based on existing user_type
UPDATE public.users 
SET role = CASE 
  WHEN user_type = 'lab' THEN 'lab'
  WHEN user_type = 'admin' THEN 'central_admin'
  ELSE 'patient'
END
WHERE role IS NULL;

-- 3. Set default role for new users
ALTER TABLE public.users ALTER COLUMN role SET DEFAULT 'patient';

-- 4. Add constraint to ensure valid roles
ALTER TABLE public.users DROP CONSTRAINT IF EXISTS valid_role;
ALTER TABLE public.users ADD CONSTRAINT valid_role 
CHECK (role IN ('patient', 'lab', 'central_admin'));

-- 5. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON public.users(role);
CREATE INDEX IF NOT EXISTS idx_users_lab_name ON public.users(lab_name);

-- 6. Enable RLS (if not already enabled)
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- 7. Drop existing policies to recreate them
DROP POLICY IF EXISTS "Users can view own profile" ON public.users;
DROP POLICY IF EXISTS "Users can update own profile" ON public.users;
DROP POLICY IF EXISTS "Users can insert own profile" ON public.users;
DROP POLICY IF EXISTS "Service role can manage all users" ON public.users;
DROP POLICY IF EXISTS "Authenticated users can insert own profile" ON public.users;
DROP POLICY IF EXISTS "Admins can view all users" ON public.users;

-- 8. Create RLS policies for the application
-- Policy: Users can view their own profile
CREATE POLICY "Users can view own profile"
ON public.users
FOR SELECT
USING (id = auth.uid());

-- Policy: Users can update their own profile
CREATE POLICY "Users can update own profile"
ON public.users
FOR UPDATE
USING (id = auth.uid());

-- Policy: Allow inserting own profile during signup
-- This is the key policy that enables signup to work
CREATE POLICY "Authenticated users can insert own profile"
ON public.users
FOR INSERT
WITH CHECK (id = auth.uid());

-- Policy: Admins can view all users (for admin dashboard)
CREATE POLICY "Admins can view all users"
ON public.users
FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.users u 
    WHERE u.id = auth.uid() 
    AND (u.role = 'central_admin' OR u.user_type = 'admin')
  )
);

-- 9. Function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 10. Trigger to update updated_at on row update
DROP TRIGGER IF EXISTS users_updated_at ON public.users;
CREATE TRIGGER users_updated_at
  BEFORE UPDATE ON public.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_updated_at();

-- ============================================================================
-- VERIFY SETUP
-- ============================================================================
SELECT 'Users table updated successfully' as status;

-- Show table structure
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'users' AND table_schema = 'public'
ORDER BY ordinal_position;

-- Show existing users
SELECT id, email, role, lab_name, user_type, created_at 
FROM public.users 
ORDER BY created_at DESC 
LIMIT 10;